// takes in starting pointer to 32 nvfp4 elements of a_frag and b_frag, 8b float scale factors packed
// converts nvfp4 -> fp16, multiplies and sums them (results in two sums), then multiplies two resultant 
// sums by correct scale factors, adds that result to value in thread-local reg accum
__device__ void elem_dot_32_(const at::Float4_e2m1fnx2* __restrict__ a_frag, const at::Float4_e2m1fnx2* __restrict__ b_frag, 
                            const at::Float8_e4m3fn* __restrict__ sfa_frag, const at::Float8_e4m3fn* __restrict__ sfb_frag, at::Half* accum) {

    uint16_t const& sfa_frag_packed = reinterpret_cast<uint16_t const&>(sfa_frag);
    uint16_t const& sfb_frag_packed = reinterpret_cast<uint16_t const&>(sfb_frag);

    uint32_t const* a_frag_packed = reinterpret_cast<uint32_t const*>(a_frag);
    uint32_t const* b_frag_packed = reinterpret_cast<uint32_t const*>(b_frag);

    uint16_t* accum_fp16 = reinterpret_cast<uint16_t*>(accum);
    uint16_t res = 0;
    uint16_t* out_fp16 = &res;

	asm volatile( \
            "{\n" \
            "// declare registers for A / B tensors\n" \
            ".reg .b8 byte0_0, byte0_1, byte0_2, byte0_3;\n" \
            ".reg .b8 byte0_4, byte0_5, byte0_6, byte0_7;\n" \
            ".reg .b8 byte1_0, byte1_1, byte1_2, byte1_3;\n" \
            ".reg .b8 byte1_4, byte1_5, byte1_6, byte1_7;\n" \
            ".reg .b8 byte2_0, byte2_1, byte2_2, byte2_3;\n" \
            ".reg .b8 byte2_4, byte2_5, byte2_6, byte2_7;\n" \
            ".reg .b8 byte3_0, byte3_1, byte3_2, byte3_3;\n" \
            ".reg .b8 byte3_4, byte3_5, byte3_6, byte3_7;\n" \

            "// declare registers for accumulators\n" \
            ".reg .f16x2 accum_0_0, accum_0_1, accum_0_2, accum_0_3;\n" \
            ".reg .f16x2 accum_1_0, accum_1_1, accum_1_2, accum_1_3;\n" \
            ".reg .f16x2 accum_2_0, accum_2_1, accum_2_2, accum_2_3;\n" \
            ".reg .f16x2 accum_3_0, accum_3_1, accum_3_2, accum_3_3;\n" \

            "// declare registers for scaling factors\n" \
            ".reg .f16x2 sfa_f16x2;\n" \
            ".reg .f16x2 sfb_f16x2;\n" \
            ".reg .f16x2 sf_f16x2;\n" \
            
            "// declare registers for conversion\n" \
            ".reg .f16x2 cvt_0_0, cvt_0_1, cvt_0_2, cvt_0_3;\n" \
            ".reg .f16x2 cvt_0_4, cvt_0_5, cvt_0_6, cvt_0_7;\n" \
            ".reg .f16x2 cvt_1_0, cvt_1_1, cvt_1_2, cvt_1_3;\n" \
            ".reg .f16x2 cvt_1_4, cvt_1_5, cvt_1_6, cvt_1_7;\n" \
            ".reg .f16x2 cvt_2_0, cvt_2_1, cvt_2_2, cvt_2_3;\n" \
            ".reg .f16x2 cvt_2_4, cvt_2_5, cvt_2_6, cvt_2_7;\n" \
            ".reg .f16x2 cvt_3_0, cvt_3_1, cvt_3_2, cvt_3_3;\n" \
            ".reg .f16x2 cvt_3_4, cvt_3_5, cvt_3_6, cvt_3_7;\n" \
            ".reg .f16 result_f16, lane0, lane1;\n" \
            ".reg .f16x2 mul_f16x2_0, mul_f16x2_1;\n" \

            "// convert scaling factors from fp8 to f16x2\n" \
            "cvt.rn.f16x2.e4m3x2 sfa_f16x2, %1;\n" \
            "cvt.rn.f16x2.e4m3x2 sfb_f16x2, %2;\n" \
            
            "// clear accumulators\n" \
            "mov.b32 accum_0_0, 0;\n" \
            "mov.b32 accum_0_1, 0;\n" \
            "mov.b32 accum_0_2, 0;\n" \
            "mov.b32 accum_0_3, 0;\n" \
            "mov.b32 accum_1_0, 0;\n" \
            "mov.b32 accum_1_1, 0;\n" \
            "mov.b32 accum_1_2, 0;\n" \
            "mov.b32 accum_1_3, 0;\n" \
            "mov.b32 accum_2_0, 0;\n" \
            "mov.b32 accum_2_1, 0;\n" \
            "mov.b32 accum_2_2, 0;\n" \
            "mov.b32 accum_2_3, 0;\n" \
            "mov.b32 accum_3_0, 0;\n" \
            "mov.b32 accum_3_1, 0;\n" \
            "mov.b32 accum_3_2, 0;\n" \
            "mov.b32 accum_3_3, 0;\n" \
            
            "// multiply, unpacking and permuting scale factors\n" \
            "mul.rn.f16x2 sf_f16x2, sfa_f16x2, sfb_f16x2;\n" \
            "mov.b32 {lane0, lane1}, sf_f16x2;\n" \
            "mov.b32 mul_f16x2_0, {lane0, lane0};\n" \
            "mov.b32 mul_f16x2_1, {lane1, lane1};\n" \

            "// unpacking A and B tensors\n" \
            "mov.b32 {byte0_0, byte0_1, byte0_2, byte0_3}, %3;\n" \
            "mov.b32 {byte0_4, byte0_5, byte0_6, byte0_7}, %4;\n" \
            "mov.b32 {byte1_0, byte1_1, byte1_2, byte1_3}, %5;\n" \
            "mov.b32 {byte1_4, byte1_5, byte1_6, byte1_7}, %6;\n" \
            "mov.b32 {byte2_0, byte2_1, byte2_2, byte2_3}, %7;\n" \
            "mov.b32 {byte2_4, byte2_5, byte2_6, byte2_7}, %8;\n" \
            "mov.b32 {byte3_0, byte3_1, byte3_2, byte3_3}, %9;\n" \
            "mov.b32 {byte3_4, byte3_5, byte3_6, byte3_7}, %10;\n" \

            "// convert A and B tensors from fp4 to f16x2\n" \

            "// A[0 - 7] and B[0 - 7]\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_0_0, byte0_0;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_0_1, byte0_1;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_0_2, byte0_2;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_0_3, byte0_3;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_0_4, byte0_4;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_0_5, byte0_5;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_0_6, byte0_6;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_0_7, byte0_7;\n" \

            "// A[8 - 15] and B[8 - 15]\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_1_0, byte1_0;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_1_1, byte1_1;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_1_2, byte1_2;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_1_3, byte1_3;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_1_4, byte1_4;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_1_5, byte1_5;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_1_6, byte1_6;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_1_7, byte1_7;\n" \

            "// A[16 - 23] and B[16 - 23]\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_2_0, byte2_0;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_2_1, byte2_1;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_2_2, byte2_2;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_2_3, byte2_3;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_2_4, byte2_4;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_2_5, byte2_5;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_2_6, byte2_6;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_2_7, byte2_7;\n" \

            "// A[24 - 31] and B[24 - 31]\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_3_0, byte3_0;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_3_1, byte3_1;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_3_2, byte3_2;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_3_3, byte3_3;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_3_4, byte3_4;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_3_5, byte3_5;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_3_6, byte3_6;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_3_7, byte3_7;\n" \

            "// fma for A[0 - 7] and B[0 - 7]\n" \
            "fma.rn.f16x2 accum_0_0, cvt_0_0, cvt_0_4, accum_0_0;\n" \
            "fma.rn.f16x2 accum_0_1, cvt_0_1, cvt_0_5, accum_0_1;\n" \
            "fma.rn.f16x2 accum_0_2, cvt_0_2, cvt_0_6, accum_0_2;\n" \
            "fma.rn.f16x2 accum_0_3, cvt_0_3, cvt_0_7, accum_0_3;\n" \

            "// fma for A[8 - 15] and B[8 - 15]\n" \
            "fma.rn.f16x2 accum_1_0, cvt_1_0, cvt_1_4, accum_1_0;\n" \
            "fma.rn.f16x2 accum_1_1, cvt_1_1, cvt_1_5, accum_1_1;\n" \
            "fma.rn.f16x2 accum_1_2, cvt_1_2, cvt_1_6, accum_1_2;\n" \
            "fma.rn.f16x2 accum_1_3, cvt_1_3, cvt_1_7, accum_1_3;\n" \

            "// fma for A[16 - 23] and B[16 - 23\n]" \
            "fma.rn.f16x2 accum_2_0, cvt_2_0, cvt_2_4, accum_2_0;\n" \
            "fma.rn.f16x2 accum_2_1, cvt_2_1, cvt_2_5, accum_2_1;\n" \
            "fma.rn.f16x2 accum_2_2, cvt_2_2, cvt_2_6, accum_2_2;\n" \
            "fma.rn.f16x2 accum_2_3, cvt_2_3, cvt_2_7, accum_2_3;\n" \

            "// fma for A[24 - 31] and B[24 - 31]\n" \
            "fma.rn.f16x2 accum_3_0, cvt_3_0, cvt_3_4, accum_3_0;\n" \
            "fma.rn.f16x2 accum_3_1, cvt_3_1, cvt_3_5, accum_3_1;\n" \
            "fma.rn.f16x2 accum_3_2, cvt_3_2, cvt_3_6, accum_3_2;\n" \
            "fma.rn.f16x2 accum_3_3, cvt_3_3, cvt_3_7, accum_3_3;\n" \

            "// tree reduction for accumulators\n" \
            "add.rn.f16x2 accum_0_0, accum_0_0, accum_0_1;\n" \
            "add.rn.f16x2 accum_0_2, accum_0_2, accum_0_3;\n" \
            "add.rn.f16x2 accum_1_0, accum_1_0, accum_1_1;\n" \
            "add.rn.f16x2 accum_1_2, accum_1_2, accum_1_3;\n" \
            "add.rn.f16x2 accum_2_0, accum_2_0, accum_2_1;\n" \
            "add.rn.f16x2 accum_2_2, accum_2_2, accum_2_3;\n" \
            "add.rn.f16x2 accum_3_0, accum_3_0, accum_3_1;\n" \
            "add.rn.f16x2 accum_3_2, accum_3_2, accum_3_3;\n" \

            "add.rn.f16x2 accum_0_0, accum_0_0, accum_0_2;\n" \
            "add.rn.f16x2 accum_1_0, accum_1_0, accum_1_2;\n" \
            "add.rn.f16x2 accum_2_0, accum_2_0, accum_2_2;\n" \
            "add.rn.f16x2 accum_3_0, accum_3_0, accum_3_2;\n" \

            "add.rn.f16x2 accum_0_0, accum_0_0, accum_1_0;\n" \
            "add.rn.f16x2 accum_2_0, accum_2_0, accum_3_0;\n" \

            "// apply scaling factors and final reduction\n" \
            "mul.rn.f16x2 accum_0_0, mul_f16x2_0, accum_0_0;\n" \
            "mul.rn.f16x2 accum_2_0, mul_f16x2_1, accum_2_0;\n" \

            "add.rn.f16x2 accum_0_0, accum_0_0, accum_2_0;\n" \
            
            "mov.b32 {lane0, lane1}, accum_0_0;\n" \
            "add.rn.f16 result_f16, lane0, lane1;\n" \

            "mov.b16 %0, result_f16;\n" \

            "}\n" \
            : "=h"(out_fp16[0])                                     // 0
            : "h"(sfa_frag_packed), "h"(sfb_frag_packed),     // 1, 2
              "r"(a_frag_packed[0]), "r"(b_frag_packed[0]),   // 3, 4
              "r"(a_frag_packed[1]), "r"(b_frag_packed[1]),   // 5, 6
              "r"(a_frag_packed[2]), "r"(b_frag_packed[2]),   // 7, 8
              "r"(a_frag_packed[3]), "r"(b_frag_packed[3])    // 9, 10
            : "memory"
        );
    *accum_fp16 += res;
}
__global__ void batched_nvfp4_gemv_kernel(const at::Float4_e2m1fnx2* __restrict__ a_ref, const at::Float4_e2m1fnx2* __restrict__ b_ref, 
                                          const at::Float8_e4m3fn* __restrict__ sfa_ref, const at::Float8_e4m3fn* __restrict__ sfb_ref, 
                                          at::Half* __restrict__ c_ref, int m, int k, int l) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = tid / m;
    int row = tid % m;
    int a_base = batch*m*(k/2) + row*(k/2);
    int b_base = batch*(k/2);
    int sfa_base = batch*m*(k/16) + row*(k/16);
    int sfb_base = batch*(k/16);
    int c_base = batch*m + row;

    at::Half accum = 0;

    for (int frag = 0; frag < k; frag += 32) {
        elem_dot_32_(&a_ref[a_base+(frag/2)], &b_ref[b_base+(frag/2)], &sfa_ref[sfa_base+(frag/16)], &sfb_ref[sfb_base+(frag/16)], &accum);
    }
    c_ref[c_base] = accum;
}

torch::Tensor batched_nvfp4_gemv(torch::Tensor a_ref, torch::Tensor b_ref, torch::Tensor sfa_ref, torch::Tensor sfb_ref, 
                                 torch::Tensor c_ref, int m, int k, int l) { 
    const int threads = 256; 
    const int blocks = (m*k + threads - 1) / threads;  
    
    // In v1 each thread computes it's own block-scaled dot-product
    // There are M*L dot-products
    batched_nvfp4_gemv_kernel<<<blocks, threads>>>(a_ref.data_ptr<at::Float4_e2m1fnx2>(), b_ref.data_ptr<at::Float4_e2m1fnx2>(), 
                                                   sfa_ref.data_ptr<at::Float8_e4m3fn>(), sfb_ref.data_ptr<at::Float8_e4m3fn>(), 
                                                   c_ref.data_ptr<at::Half>(), m, k, l);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return c_ref;
}

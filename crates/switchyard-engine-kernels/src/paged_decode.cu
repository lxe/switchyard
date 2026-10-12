// One block owns one row/query head. Pages stay compact; there is no padded
// history, dense materialization or CPU attention computation.
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

__device__ float warp_sum(float x) {
    for (int offset = 16; offset; offset >>= 1)
        x += __shfl_down_sync(0xffffffff, x, offset);
    return x;
}
__device__ float block_sum(float x, float* scratch) {
    x = warp_sum(x);
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x / 32] = x;
    __syncthreads();
    x = threadIdx.x < 8 ? scratch[threadIdx.x] : 0.f;
    if (threadIdx.x < 32) x = warp_sum(x);
    if (threadIdx.x == 0) scratch[0] = x;
    __syncthreads();
    return scratch[0];
}
__device__ float block_max(float x, float* scratch) {
    for (int offset = 16; offset; offset >>= 1)
        x = fmaxf(x, __shfl_down_sync(0xffffffff, x, offset));
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x / 32] = x;
    __syncthreads();
    x = threadIdx.x < 8 ? scratch[threadIdx.x] : -INFINITY;
    if (threadIdx.x < 32)
        for (int offset = 16; offset; offset >>= 1)
            x = fmaxf(x, __shfl_down_sync(0xffffffff, x, offset));
    if (threadIdx.x == 0) scratch[0] = x;
    __syncthreads();
    return scratch[0];
}

template<typename T> __device__ T rounded(float x);
template<> __device__ float rounded<float>(float x) { return x; }
template<> __device__ __nv_bfloat16 rounded<__nv_bfloat16>(float x) { return __float2bfloat16_rn(x); }

template<typename T>
__device__ void paged_decode(
    const T* query, const uint64_t* pages, const uint32_t* row_offsets,
    const uint32_t* lengths, const uint32_t* page_tokens, float* scores,
    T* output, int heads, int groups, int dim, int max_len) {
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int kv_head = head / groups;
    const int length = lengths[row];
    const int page_width = page_tokens[row];
    const int page_base = row_offsets[row];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    float* score = scores + (row * heads + head) * max_len;
    __shared__ float q[256];
    __shared__ float reduction[8];
    for (int d = threadIdx.x; d < dim; d += blockDim.x)
        q[d] = float(query[(row * heads + head) * dim + d]);
    __syncthreads();
    const float scale = rsqrtf(float(dim));
    const int page_count = (length - 1) / page_width + 1;
    for (int page = 0; page < page_count; ++page) {
        const int p = page_base + page;
        const int live = int(pages[p * 3 + 2]);
        const T* key = reinterpret_cast<const T*>(pages[p * 3]);
        const int head_base = kv_head * live * dim;
        for (int token = warp; token < live; token += 8) {
            float dot = 0.f;
            for (int d = lane; d < dim; d += 32)
                dot = fmaf(q[d], float(key[head_base + token * dim + d]), dot);
            dot = warp_sum(dot);
            if (lane == 0) score[page * page_width + token] = dot * scale;
        }
    }
    __syncthreads();
    float maximum = -INFINITY;
    for (int token = threadIdx.x; token < length; token += blockDim.x)
        maximum = fmaxf(maximum, score[token]);
    maximum = block_max(maximum, reduction);
    float denominator = 0.f;
    for (int token = threadIdx.x; token < length; token += blockDim.x) {
        const float exponential = expf(score[token] - maximum);
        denominator += exponential;
        // Reference BF16 attention rounds only the numerator's exponentials.
        score[token] = float(rounded<T>(exponential));
    }
    denominator = block_sum(denominator, reduction);
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float accumulator = 0.f;
        for (int page = 0; page < page_count; ++page) {
            const int p = page_base + page;
            const int live = int(pages[p * 3 + 2]);
            const T* value = reinterpret_cast<const T*>(pages[p * 3 + 1]);
            const int head_base = kv_head * live * dim;
            const int score_base = page * page_width;
            for (int token = 0; token < live; ++token)
                accumulator = fmaf(score[score_base + token], float(value[head_base + token * dim + d]), accumulator);
        }
        output[(row * heads + head) * dim + d] = rounded<T>(accumulator / denominator);
    }
}

extern "C" __global__ void paged_decode_f32(const float* q, const uint64_t* p, const uint32_t* r, const uint32_t* l, const uint32_t* w, float* s, float* o, int h, int g, int d, int m) { paged_decode(q,p,r,l,w,s,o,h,g,d,m); }
extern "C" __global__ void paged_decode_bf16(const __nv_bfloat16* q, const uint64_t* p, const uint32_t* r, const uint32_t* l, const uint32_t* w, float* s, __nv_bfloat16* o, int h, int g, int d, int m) { paged_decode(q,p,r,l,w,s,o,h,g,d,m); }
